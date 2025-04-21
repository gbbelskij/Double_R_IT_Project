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
  imageSrc: string;
}
