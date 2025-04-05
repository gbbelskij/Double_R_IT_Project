import { Course } from "types/course";

export interface CourseSectionProps {
  /**
   * Array of course objects to be displayed in the section
   */
  courses: Course[];
  /**
   * Title of the course section
   */
  title: string;
}
