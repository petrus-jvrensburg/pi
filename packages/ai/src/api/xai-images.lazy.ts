import type { ProviderImages } from "../types.ts";

export const xaiImagesApi = (): ProviderImages => ({
	generateImages: async (model, context, options) =>
		(await import("./xai-images.ts")).generateImages(model, context, options),
});
