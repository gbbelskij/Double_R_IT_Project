import { GetDominantColorOptions } from "@utils/getDominantColorFromImage";

export interface CourseProps {
  /**
   * Title of the course
   */
  title: string;
  /**
   * Duration of the course in hours
   */
  duration: number;
  /**
   * Brief description of the course
   */
  description: string;
  /**
   * URL linking to the course page
   */
  url: string;
  /**
   * Image source URL for the course thumbnail
   */
  imageUrl: string;
  /**
   * Options for getting the dominant color from the image
   */
  colorOptions?: GetDominantColorOptions;
}
