import { useState } from "react";

import Course from "@components/Course/Course";
import Button from "@components/Button/Button";

import { CourseSectionProps } from "./CourseSection.props";

import classes from "./CourseSection.module.css";

const COURSES_PER_PAGE = 3;

const CourseSection: React.FC<CourseSectionProps> = ({ courses, title }) => {
  const [visibleCourses, setVisibleCourses] = useState(COURSES_PER_PAGE);

  const handleShowMore = () => {
    setVisibleCourses((prev) => prev + COURSES_PER_PAGE);
  };

  const isMoreCoursesAvailable = visibleCourses < courses.length;

  return (
    <>
      <section className={classes.CourseSection}>
        <h2 className={classes.CourseSectionTitle}>{title}</h2>

        <div className={classes.CourseSectionCards}>
          {courses.slice(0, visibleCourses).map((course) => (
            <Course key={course.id} {...course} />
          ))}
        </div>

        {isMoreCoursesAvailable && (
          <div className={classes.CourseSectionButtonWrapper}>
            <Button type="button" color="dim" onClick={handleShowMore}>
              Больше курсов
            </Button>
          </div>
        )}
      </section>
    </>
  );
};

export default CourseSection;
