import { FastAverageColor } from "fast-average-color";
import chroma from "chroma-js";

export interface GetDominantColorOptions {
  left?: number;
  top?: number;
  width?: number;
  height?: number;
}

/**
 * Gets the dominant color of the image or part of it
 * @param src image URL
 * @param options Image area selection options
 */
export const getDominantColorFromImage = (
  src: string,
  options: GetDominantColorOptions = {}
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = src;

    img
      .decode()
      .then(() => {
        const fac = new FastAverageColor();

        const {
          left = 0,
          top = 0,
          width = img.width,
          height = img.height,
        } = options;
        fac
          .getColorAsync(img, {
            left,
            top,
            width,
            height,
          })
          .then((color) => {
            const vibrantColor = chroma(color.hex)
              .set("hsl.l", 0.5)
              .saturate(2)
              .hex();
            resolve(vibrantColor);
          })
          .catch(reject);
      })
      .catch(reject);
  });
};
