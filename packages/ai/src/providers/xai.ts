import { openAIResponsesApi } from "../api/openai-responses.lazy.ts";
import { xaiImagesApi } from "../api/xai-images.lazy.ts";
import { envApiKeyAuth, lazyOAuth } from "../auth/helpers.ts";
import { loadXaiOAuth } from "../auth/oauth/load.ts";
import { createProvider, type Provider } from "../models.ts";
import { XAI_IMAGE_MODELS, XAI_MODELS } from "./xai.models.ts";

export function xaiProvider(): Provider<"openai-responses"> {
	return createProvider({
		id: "xai",
		name: "xAI",
		baseUrl: "https://api.x.ai/v1",
		auth: {
			apiKey: envApiKeyAuth("xAI API key", ["XAI_API_KEY"]),
			oauth: lazyOAuth({
				name: "xAI (Grok/X subscription)",
				isSubscription: true,
				loginLabel: "Sign in with SuperGrok or X Premium",
				load: loadXaiOAuth,
			}),
		},
		models: [...Object.values(XAI_MODELS), ...Object.values(XAI_IMAGE_MODELS)],
		api: openAIResponsesApi(),
		images: { "xai-images": xaiImagesApi() },
	});
}
