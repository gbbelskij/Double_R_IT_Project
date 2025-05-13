import { RefObject } from "react";

export interface BackgroundElementsProps {
  /**
   * @todo
   */
  targetRef?: RefObject<null | HTMLElement>;
  /**
   * @todo
   */
  color?: string;
  /**
   * @todo
   */
  count?: number;
  /**
   * @todo
   */
  blobsExtraPoints?: number;
  /**
   * @todo
   */
  blobsRandomness?: number;
  /**
   * @todo
   */
  blobsSize?: number;
  /**
   * @todo
   */
  styles?: string;
}
