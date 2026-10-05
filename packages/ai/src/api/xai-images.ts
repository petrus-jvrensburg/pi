import type {
	AssistantImages,
	ImageApi,
	ImageContent,
	ImageModel,
	ImagesContext,
	ImagesFunction,
	ImagesOptions,
	ProviderHeaders,
	Usage,
} from "../types.ts";
import { formatProviderError, normalizeProviderError, truncateErrorText } from "../utils/error-body.ts";
import { headersToRecord, providerHeadersToRecord } from "../utils/headers.ts";
import { retryProviderRequest } from "../utils/provider-retry.ts";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.ts";

/**
 * Edit cap. Release notes say 5. The multi-image guide still says 3, and the edits schema has no maxItems.
 * https://docs.x.ai/developers/release-notes
 * https://docs.x.ai/developers/model-capabilities/images/multi-image-editing
 */
const MAX_SOURCE_IMAGES = 5;

/** https://docs.x.ai/developers/models — grok-imagine-image-2.0 is $0.04 / image. */
const USD_PER_IMAGE = 0.04;
/** One US dollar equals 10,000,000,000 ticks. https://docs.x.ai/developers/rest-api-reference/inference/images */
const USD_TICKS_PER_DOLLAR = 10_000_000_000;

const SUPPORTED_SOURCE_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

/**
 * Generation documents at most 10 images per request. The edit reference calls `n`
 * the number of image edits and states no maximum, so edits are not capped.
 * https://docs.x.ai/developers/model-capabilities/imagine
 * https://docs.x.ai/developers/rest-api-reference/inference/images
 */
const MAX_GENERATION_IMAGES = 10;

const ASPECT_RATIOS = [
	"1:1",
	"3:4",
	"4:3",
	"9:16",
	"16:9",
	"2:3",
	"3:2",
	"9:19.5",
	"19.5:9",
	"9:20",
	"20:9",
	"1:2",
	"2:1",
	"21:9",
	"5:2",
	"auto",
] as const;
const RESOLUTIONS = ["1k", "2k"] as const;
/**
 * Omitted quality keeps the service default (`auto`: low for generation, medium for editing).
 * https://docs.x.ai/developers/release-notes
 */
const QUALITIES = ["auto", "low", "medium"] as const;

type AspectRatio = (typeof ASPECT_RATIOS)[number];
type Resolution = (typeof RESOLUTIONS)[number];
type Quality = (typeof QUALITIES)[number];

interface XaiImagesHttpError extends Error {
	status: number | undefined;
	headers: Headers | undefined;
	body?: string;
}

interface XaiImageRef {
	type: "image_url";
	url: string;
}

interface XaiImagesRequest {
	model: string;
	prompt: string;
	response_format: "b64_json";
	image?: XaiImageRef;
	images?: XaiImageRef[];
	aspect_ratio?: AspectRatio;
	resolution?: Resolution;
	quality?: Quality;
	n?: number;
}

/** Image generation and editing on xAI's Imagine REST API. */
export const generateImages: ImagesFunction<ImagesOptions> = async (
	model: ImageModel<ImageApi>,
	context: ImagesContext,
	options?: ImagesOptions,
) => {
	const output: AssistantImages = {
		api: model.api,
		provider: model.provider,
		model: model.id,
		output: [],
		stopReason: "stop",
		timestamp: Date.now(),
	};

	try {
		const apiKey = options?.apiKey;
		if (!apiKey) {
			throw new Error(`No API key for provider: ${model.provider}`);
		}
		const images = sourceImages(context);
		let payload: unknown = buildRequest(model, context, images, options?.metadata);
		const nextPayload = await options?.onPayload?.(payload, model);
		if (nextPayload !== undefined) payload = nextPayload;

		const requestFetch = options?.fetch ?? globalThis.fetch;
		const url = endpoint(model.baseUrl, images.length > 0 ? "/images/edits" : "/images/generations");
		const { response, body } = await retryProviderRequest(
			async () => {
				const timeoutSignal = options?.timeoutMs !== undefined ? AbortSignal.timeout(options.timeoutMs) : undefined;
				const signal =
					options?.signal && timeoutSignal
						? AbortSignal.any([options.signal, timeoutSignal])
						: (options?.signal ?? timeoutSignal);
				try {
					const next = await requestFetch(url, {
						method: "POST",
						headers: requestHeaders(model, apiKey, options?.headers),
						body: JSON.stringify(payload),
						signal,
					});
					if (!next.ok) throw httpError(next, await next.text());
					return { response: next, body: (await next.json()) as unknown };
				} catch (error) {
					if (timeoutSignal?.aborted && !options?.signal?.aborted) throw timeoutError(options.timeoutMs!);
					throw error;
				}
			},
			{
				maxRetries: options?.maxRetries,
				maxRetryDelayMs: options?.maxRetryDelayMs,
				signal: options?.signal,
			},
		);
		await options?.onResponse?.({ status: response.status, headers: headersToRecord(response.headers) }, model);

		if (!isRecord(body)) throw new Error("xAI returned an unexpected image response");
		if (typeof body.id === "string") output.responseId = body.id;
		const blocks = imageBlocks(body);
		if (blocks.length === 0) {
			throw new Error(hasImageUrl(body) ? "xAI returned an image URL instead of base64" : "xAI returned no image");
		}
		output.output.push(...blocks);
		output.usage = parseUsage(body.usage, images.length, blocks.length);
		return output;
	} catch (error) {
		output.stopReason = options?.signal?.aborted ? "aborted" : "error";
		output.errorMessage = formatProviderError(normalizeProviderError(error));
		return output;
	}
};

function endpoint(baseUrl: string, path: string): string {
	return `${baseUrl.replace(/\/+$/u, "")}${path}`;
}

function requestHeaders(
	model: ImageModel<ImageApi>,
	apiKey: string,
	optionsHeaders?: ProviderHeaders,
): Record<string, string> {
	return (
		providerHeadersToRecord(
			{ authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
			model.headers,
			optionsHeaders,
		) ?? {}
	);
}

function sourceImages(context: ImagesContext): XaiImageRef[] {
	const images = context.input.filter((item) => item.type === "image");
	if (images.length > MAX_SOURCE_IMAGES) {
		throw new Error(`xAI image edits accept at most ${MAX_SOURCE_IMAGES} source images`);
	}
	return images.map((item) => {
		if (!SUPPORTED_SOURCE_MIME_TYPES.has(item.mimeType)) {
			throw new Error(`xAI image edits accept JPEG, PNG, or WebP, not ${item.mimeType}`);
		}
		return {
			type: "image_url",
			url: `data:${item.mimeType};base64,${item.data}`,
		};
	});
}

/**
 * Join text blocks as the prompt. Do not insert `<IMAGE_0>`; xAI's edit guide uses ordinary language.
 * That token is an optional caller-written index into image blocks, not content order.
 * One image uses `image` and needs no index. The first image sets the default aspect ratio.
 * https://docs.x.ai/developers/rest-api-reference/inference/images
 * https://docs.x.ai/developers/model-capabilities/images/multi-image-editing
 */
function buildRequest(
	model: ImageModel<ImageApi>,
	context: ImagesContext,
	images: XaiImageRef[],
	metadata: Record<string, unknown> | undefined,
): XaiImagesRequest {
	const prompt = context.input
		.filter((item) => item.type === "text")
		.map((item) => sanitizeSurrogates(item.text))
		.join("\n")
		.trim();
	if (prompt.length === 0) throw new Error("xAI image generation requires a text prompt");

	const request: XaiImagesRequest = {
		model: model.id,
		prompt,
		response_format: "b64_json",
	};
	if (images.length === 1) request.image = images[0];
	else if (images.length > 1) request.images = images;
	// Before onPayload, so a caller can still replace these fields.
	applyImageControls(request, metadata, images.length > 0);
	return request;
}

/** Copy only the controls this API understands. Other metadata keys stay off the request. */
function applyImageControls(
	request: XaiImagesRequest,
	metadata: Record<string, unknown> | undefined,
	editing: boolean,
): void {
	if (!metadata) return;
	const aspectRatio = enumField(metadata, "aspect_ratio", ASPECT_RATIOS);
	if (aspectRatio !== undefined) request.aspect_ratio = aspectRatio;
	const resolution = enumField(metadata, "resolution", RESOLUTIONS);
	if (resolution !== undefined) request.resolution = resolution;
	const quality = enumField(metadata, "quality", QUALITIES);
	if (quality !== undefined) request.quality = quality;
	const count = imageCount(metadata, editing);
	if (count !== undefined) request.n = count;
}

function enumField<T extends string>(
	metadata: Record<string, unknown>,
	key: string,
	allowed: readonly T[],
): T | undefined {
	if (!Object.hasOwn(metadata, key) || metadata[key] === undefined) return undefined;
	const value = metadata[key];
	if (typeof value === "string" && allowed.includes(value as T)) return value as T;
	// Do not echo the value: it may be a URL or image bytes.
	throw new Error(`xAI ${key} is not a supported value`);
}

function imageCount(metadata: Record<string, unknown>, editing: boolean): number | undefined {
	if (!Object.hasOwn(metadata, "n") || metadata.n === undefined) return undefined;
	const value = metadata.n;
	const withinLimit = typeof value === "number" && Number.isInteger(value) && value >= 1;
	if (withinLimit && (editing || value <= MAX_GENERATION_IMAGES)) return value;
	throw new Error(
		editing
			? "xAI n must be an integer greater than or equal to 1"
			: `xAI n must be an integer from 1 to ${MAX_GENERATION_IMAGES}`,
	);
}

function imageBlocks(body: Record<string, unknown>): ImageContent[] {
	if (!Array.isArray(body.data)) return [];
	const blocks: ImageContent[] = [];
	for (const item of body.data) {
		if (!isRecord(item) || typeof item.b64_json !== "string" || item.b64_json.length === 0) continue;
		const mimeType =
			typeof item.mime_type === "string" && item.mime_type.startsWith("image/") ? item.mime_type : "image/png";
		blocks.push({ type: "image", mimeType, data: item.b64_json });
	}
	return blocks;
}

function hasImageUrl(body: Record<string, unknown>): boolean {
	return Array.isArray(body.data) && body.data.some((item) => isRecord(item) && typeof item.url === "string");
}

function parseUsage(raw: unknown, sourceImages: number, outputImages: number): Usage {
	// Imagine bills per image, not per token. Token counts stay zero. The request
	// price is recorded on output so session cost can add cost.total.
	const usage = isRecord(raw) ? raw : undefined;
	const ticks = usage?.cost_in_usd_ticks;
	const dollars =
		typeof ticks === "number" && Number.isFinite(ticks)
			? ticks / USD_TICKS_PER_DOLLAR
			: USD_PER_IMAGE * (sourceImages + outputImages);
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: dollars, cacheRead: 0, cacheWrite: 0, total: dollars },
	};
}

function httpError(response: Response, body: string): XaiImagesHttpError {
	const error = new Error(`xAI image generation returned ${response.status}`) as XaiImagesHttpError;
	error.status = response.status;
	error.headers = response.headers;
	error.body = truncateErrorText(body, 4000);
	return error;
}

function timeoutError(timeoutMs: number): XaiImagesHttpError {
	const error = new Error(`Request timed out after ${timeoutMs}ms`) as XaiImagesHttpError;
	error.name = "TimeoutError";
	error.status = undefined;
	error.headers = undefined;
	return error;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
