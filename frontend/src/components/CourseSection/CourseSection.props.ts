import { Course } from "../../data";

/**
 * Props for the CourseSection component
 */
export interface CourseSectionProps {
  /** Array of courses to display */
  courses: Course[];
  /** Title of the section */
  title: string;
}