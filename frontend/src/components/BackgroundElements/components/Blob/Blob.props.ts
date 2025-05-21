export interface BlobProps {
  /**
   * The color of the blob.
   */
  color?: string;
  /**
   * The seed for generating the blob's shape.
   */
  seed?: number;
  /**
   * The number of extra points for the blob's shape.
   */
  extraPoints?: number;
  /**
   * The degree of randomness in the blob's shape.
   */
  randomness?: number;
  /**
   * The size of the blob.
   */
  size?: number;
}