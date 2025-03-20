import React from "react";
import { CourseProps } from "./Course.props";
import styles from "./Course.module.css";

const Course: React.FC<CourseProps> = ({ title, duration, description, url }) => {
  return (
    <a href={url} className={styles.Course}>
      <h3 className={styles.CourseTitle}>{title}</h3>
      <p className={styles.CourseDuration}>{duration} hours</p>
      <p className={styles.CourseDescription}>{description}</p>
    </a>
  );
};

export default Course;