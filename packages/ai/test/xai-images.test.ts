import { beforeEach, describe, expect, it, vi } from "vitest";
import { generateImages } from "../src/images.ts";
import type { ImageModel, ImagesContext } from "../src/types.ts";

const imageData = "ZmFrZS1wbmc=";
const secretUrl = "https://imgen.example/secret.png";

const mockState = vi.hoisted(() => ({
	lastUrl: undefined as string | undefined,
	lastInit: undefined as RequestInit | undefined,
	responseBody: {} as unknown,
	aborted: false,
}));

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "content-type": "application/json" },
	});
}

vi.stubGlobal(
	"fetch",
	vi.fn(async (url: string, init?: RequestInit) => {
		mockState.lastUrl = url;
		mockState.lastInit = init;
		const signal = init?.signal;
		if (signal?.aborted || mockState.aborted) {
			throw new DOMException("The operation was aborted", "AbortError");
		}
		if (typeof mockState.responseBody === "function") {
			return (mockState.responseBody as () => Response)();
		}
		return jsonResponse(mockState.responseBody);
	}),
);

function model(): ImageModel<"xai-images"> {
	return {
		type: "image",
		id: "grok-imagine-image-2.0",
		name: "Grok Imagine Image 2.0",
		api: "xai-images",
		provider: "xai",
		baseUrl: "https://api.x.ai/v1",
		input: ["text", "image"],
		output: ["image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		headers: { "X-Title": "pi" },
	};
}

function requestBody(): {
	model?: string;
	prompt?: string;
	response_format?: string;
	image?: { type?: string; url?: string };
	images?: { type?: string; url?: string }[];
	aspect_ratio?: string;
	resolution?: string;
	quality?: string;
	n?: number;
} {
	return JSON.parse(String(mockState.lastInit?.body));
}

const IMAGE_CONTROLS = ["aspect_ratio", "resolution", "quality", "n"] as const;

describe("xai images", () => {
	beforeEach(() => {
		mockState.lastUrl = undefined;
		mockState.lastInit = undefined;
		mockState.aborted = false;
		mockState.responseBody = {
			id: "img-1",
			data: [{ b64_json: imageData, mime_type: "image/jpeg" }],
			usage: { cost_in_usd_ticks: 400_000_000 },
		};
	});

	it("generates from a text prompt and returns base64 image blocks", async () => {
		const log = vi.spyOn(console, "log").mockImplementation(() => {});
		const debug = vi.spyOn(console, "debug").mockImplementation(() => {});
		const info = vi.spyOn(console, "info").mockImplementation(() => {});
		const context: ImagesContext = {
			input: [{ type: "text", text: "Generate a dog" }],
		};

		const output = await generateImages(model(), context, { apiKey: "test-key" });
		expect(output.stopReason).toBe("stop");
		expect(output.responseId).toBe("img-1");
		expect(output.output).toEqual([{ type: "image", mimeType: "image/jpeg", data: imageData }]);
		expect(output.usage?.cost.total).toBeCloseTo(0.04);
		expect(output.usage?.input).toBe(0);
		expect(output.usage?.output).toBe(0);

		expect(mockState.lastUrl).toBe("https://api.x.ai/v1/images/generations");
		expect(requestBody()).toEqual({
			model: "grok-imagine-image-2.0",
			prompt: "Generate a dog",
			response_format: "b64_json",
		});
		expect(requestBody().quality).toBeUndefined();
		const headers = new Headers(mockState.lastInit?.headers);
		expect(headers.get("authorization")).toBe("Bearer test-key");
		expect(headers.get("x-title")).toBe("pi");
		expect(log).not.toHaveBeenCalled();
		expect(debug).not.toHaveBeenCalled();
		expect(info).not.toHaveBeenCalled();
		expect(JSON.stringify(output)).toContain(imageData);
		log.mockRestore();
		debug.mockRestore();
		info.mockRestore();
	});

	it("edits when the context includes an image", async () => {
		const context: ImagesContext = {
			input: [
				{ type: "text", text: "Render this as a pencil sketch" },
				{ type: "image", data: imageData, mimeType: "image/png" },
			],
		};

		const output = await generateImages(model(), context, { apiKey: "test-key" });
		expect(output.stopReason).toBe("stop");
		expect(mockState.lastUrl).toBe("https://api.x.ai/v1/images/edits");
		expect(requestBody().image).toEqual({
			type: "image_url",
			url: `data:image/png;base64,${imageData}`,
		});
		expect(requestBody().images).toBeUndefined();
		expect(requestBody().response_format).toBe("b64_json");
	});

	it("sends up to five source images and rejects a sixth", async () => {
		const images = [0, 1, 2, 3, 4].map((index) => ({
			type: "image" as const,
			data: `img-${index}`,
			mimeType: "image/webp" as const,
		}));
		const edited = await generateImages(
			model(),
			{ input: [{ type: "text", text: "Combine these" }, ...images] },
			{ apiKey: "test-key" },
		);
		expect(edited.stopReason).toBe("stop");
		expect(requestBody().image).toBeUndefined();
		expect(requestBody().images).toEqual(
			images.map((image) => ({ type: "image_url", url: `data:image/webp;base64,${image.data}` })),
		);

		mockState.lastUrl = undefined;
		const rejected = await generateImages(
			model(),
			{ input: [{ type: "text", text: "Combine these" }, ...images, images[0]] },
			{ apiKey: "test-key" },
		);
		expect(rejected.stopReason).toBe("error");
		expect(rejected.errorMessage).toContain("at most 5");
		expect(mockState.lastUrl).toBeUndefined();
	});

	it("does not return a temporary image URL", async () => {
		mockState.responseBody = { data: [{ url: secretUrl }] };
		const output = await generateImages(
			model(),
			{ input: [{ type: "text", text: "A dog" }] },
			{ apiKey: "test-key" },
		);
		expect(output.stopReason).toBe("error");
		expect(output.output).toEqual([]);
		expect(output.errorMessage).not.toContain(secretUrl);
		expect(JSON.stringify(output)).not.toContain(secretUrl);
	});

	it("omits image controls when metadata is absent", async () => {
		await generateImages(model(), { input: [{ type: "text", text: "A dog" }] }, { apiKey: "test-key" });
		for (const key of IMAGE_CONTROLS) expect(requestBody()).not.toHaveProperty(key);

		await generateImages(model(), { input: [{ type: "text", text: "A dog" }] }, { apiKey: "test-key", metadata: {} });
		for (const key of IMAGE_CONTROLS) expect(requestBody()).not.toHaveProperty(key);
	});

	it("copies understood metadata for generations and edits", async () => {
		const metadata = {
			aspect_ratio: "16:9",
			resolution: "2k",
			quality: "medium",
			n: 2,
			seed: "drop-me",
		};
		const generated = await generateImages(
			model(),
			{ input: [{ type: "text", text: "A dog" }] },
			{ apiKey: "test-key", metadata },
		);
		expect(generated.stopReason).toBe("stop");
		expect(requestBody()).toMatchObject({
			aspect_ratio: "16:9",
			resolution: "2k",
			quality: "medium",
			n: 2,
		});
		expect(requestBody()).not.toHaveProperty("seed");

		const edited = await generateImages(
			model(),
			{
				input: [
					{ type: "text", text: "Sketch this" },
					{ type: "image", data: imageData, mimeType: "image/png" },
				],
			},
			{ apiKey: "test-key", metadata },
		);
		expect(edited.stopReason).toBe("stop");
		expect(mockState.lastUrl).toBe("https://api.x.ai/v1/images/edits");
		expect(requestBody()).toMatchObject({
			aspect_ratio: "16:9",
			resolution: "2k",
			quality: "medium",
			n: 2,
		});
		expect(requestBody()).not.toHaveProperty("seed");
	});

	it("does not send an unknown metadata key", async () => {
		await generateImages(
			model(),
			{ input: [{ type: "text", text: "A dog" }] },
			{ apiKey: "test-key", metadata: { seed: "drop-me", modalities: ["image"] } },
		);
		expect(requestBody()).not.toHaveProperty("seed");
		expect(requestBody()).not.toHaveProperty("modalities");
		for (const key of IMAGE_CONTROLS) expect(requestBody()).not.toHaveProperty(key);
	});

	it("rejects invalid image controls before fetch", async () => {
		const source = { type: "image" as const, data: imageData, mimeType: "image/png" as const };
		const cases: { metadata: Record<string, unknown>; edit: boolean }[] = [
			{ metadata: { aspect_ratio: "square" }, edit: true },
			{ metadata: { aspect_ratio: `https://cdn.example/${imageData}` }, edit: true },
			{ metadata: { resolution: "4k" }, edit: true },
			{ metadata: { quality: "high" }, edit: false },
			{ metadata: { n: 1.5 }, edit: true },
			{ metadata: { n: "2" }, edit: false },
			{ metadata: { n: 0 }, edit: true },
			{ metadata: { n: 11 }, edit: false },
		];
		for (const { metadata, edit } of cases) {
			mockState.lastUrl = undefined;
			const output = await generateImages(
				model(),
				{ input: edit ? [{ type: "text", text: "A dog" }, source] : [{ type: "text", text: "A dog" }] },
				{ apiKey: "test-key", metadata },
			);
			expect(output.stopReason).toBe("error");
			expect(output.errorMessage).not.toContain(imageData);
			expect(output.errorMessage).not.toContain("https://");
			expect(mockState.lastUrl).toBeUndefined();
		}
	});

	it("does not cap edit counts that the edit API leaves unbounded", async () => {
		const output = await generateImages(
			model(),
			{
				input: [
					{ type: "text", text: "Sketch this" },
					{ type: "image", data: imageData, mimeType: "image/png" },
				],
			},
			{ apiKey: "test-key", metadata: { n: 11 } },
		);
		expect(output.stopReason).toBe("stop");
		expect(requestBody().n).toBe(11);
	});

	it("applies metadata before onPayload", async () => {
		let seen: Record<string, unknown> | undefined;
		await generateImages(
			model(),
			{ input: [{ type: "text", text: "A dog" }] },
			{
				apiKey: "test-key",
				metadata: { aspect_ratio: "16:9", n: 2, seed: "drop-me" },
				onPayload: (payload) => {
					seen = payload as Record<string, unknown>;
					return { ...(payload as Record<string, unknown>), aspect_ratio: "1:1", n: 4 };
				},
			},
		);
		expect(seen).toMatchObject({ aspect_ratio: "16:9", n: 2 });
		expect(seen).not.toHaveProperty("seed");
		expect(requestBody().aspect_ratio).toBe("1:1");
		expect(requestBody().n).toBe(4);
	});

	it("returns every image and prices returned images, not an assumed count", async () => {
		mockState.responseBody = {
			data: [
				{ b64_json: "aaa", mime_type: "image/png" },
				{ b64_json: "bbb", mime_type: "image/jpeg" },
				{ b64_json: "ccc", mime_type: "image/webp" },
			],
			usage: { cost_in_usd_ticks: 800_000_000 },
		};
		const withTicks = await generateImages(
			model(),
			{ input: [{ type: "text", text: "variations" }] },
			{ apiKey: "test-key", metadata: { n: 3 } },
		);
		expect(withTicks.output).toEqual([
			{ type: "image", mimeType: "image/png", data: "aaa" },
			{ type: "image", mimeType: "image/jpeg", data: "bbb" },
			{ type: "image", mimeType: "image/webp", data: "ccc" },
		]);
		expect(withTicks.usage?.cost.total).toBeCloseTo(0.08);

		mockState.responseBody = {
			data: [
				{ b64_json: "aaa", mime_type: "image/png" },
				{ b64_json: "bbb", mime_type: "image/jpeg" },
			],
		};
		const withoutTicks = await generateImages(
			model(),
			{
				input: [
					{ type: "text", text: "variations" },
					{ type: "image", data: "src", mimeType: "image/png" },
				],
			},
			{ apiKey: "test-key", metadata: { n: 3 } },
		);
		expect(withoutTicks.output).toHaveLength(2);
		// One source plus two returned images: not n, and not a single image.
		expect(withoutTicks.usage?.cost.total).toBeCloseTo(0.12);
	});

	it("prices a response that omits ticks per input and output image", async () => {
		mockState.responseBody = { data: [{ b64_json: imageData, mime_type: "image/png" }] };
		const output = await generateImages(
			model(),
			{
				input: [
					{ type: "text", text: "Sketch" },
					{ type: "image", data: "a", mimeType: "image/png" },
					{ type: "image", data: "b", mimeType: "image/jpeg" },
				],
			},
			{ apiKey: "test-key" },
		);
		expect(output.usage?.cost.total).toBeCloseTo(0.12);
	});

	it("passes through abort signal and returns aborted result", async () => {
		const controller = new AbortController();
		controller.abort();
		const output = await generateImages(
			model(),
			{ input: [{ type: "text", text: "Generate a dog" }] },
			{ apiKey: "test-key", signal: controller.signal },
		);
		expect(output.stopReason).toBe("aborted");
		expect(output.errorMessage).toBe("Request aborted");
		expect(mockState.lastInit?.signal).toBe(controller.signal);
	});
});
