export interface Course {
  /**
   * Unique identifier for the course.
   */
  id: number;
  /**
   * The title of the course.
   */
  title: string;
  /**
   * Duration of the course (in months).
   */
  duration: number;
  /**
   * A brief description of the course content.
   */
  description: string;
  /**
   * URL link to access or view the course.
   */
  url: string;
  /**
   * Source URL for the course's thumbnail or promotional image.
   */
  imageUrl: string;
}
