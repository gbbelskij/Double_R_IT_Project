import { RefObject } from "react";

export interface BackgroundElementsProps {
  /**
   * A reference to the element to which the background elements are attached.
   */
  targetRef?: RefObject<null | HTMLElement>;
  /**
   * The color of the background elements.
   */
  color?: string;
  /**
   * The number of background elements.
   */
  count?: number;
  /**
   * The number of extra points for the blobs' shapes.
   */
  blobsExtraPoints?: number;
  /**
   * The degree of randomness in the blobs' shapes.
   */
  blobsRandomness?: number;
  /**
   * The size of the blobs.
   */
  blobsSize?: number;
  /**
   * Additional styles for the background elements.
   */
  styles?: string;
}
