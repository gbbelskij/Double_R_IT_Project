import React, { useState } from "react";
import { CourseSectionProps } from "./CourseSection.props";
import Course from "../Course/Course";
import Button from "../Button/Button";
import { ButtonTypes } from "../Button/enums/ButtonTypes";
import styles from "./CourseSection.module.css";

const COURSES_PER_PAGE = 3;

const CourseSection: React.FC<CourseSectionProps> = ({ courses, title }) => {
  const [visibleCourses, setVisibleCourses] = useState(COURSES_PER_PAGE);

  const handleShowMore = () => {
    setVisibleCourses((prev) => prev + COURSES_PER_PAGE);
  };

  const isMoreCoursesAvailable = visibleCourses < courses.length;

  return (
    <section className={styles.CourseSection}>
      <div className={styles.CourseSection_Grid}>
        <h2 className={styles.CourseSection_Title}>{title}</h2>
        <div className={styles.CourseSection_Cards}>
          {courses.slice(0, visibleCourses).map((course) => (
            <Course key={course.id} {...course} />
          ))}
        </div>
        {isMoreCoursesAvailable && (
          <div className={styles.CourseSection_ButtonWrapper}>
<Button type={ButtonTypes.Default} className={styles.ShowMoreButton}>
  Больше курсов
</Button>
          </div>
        )}
      </div>
    </section>
  );
};

export default CourseSection;