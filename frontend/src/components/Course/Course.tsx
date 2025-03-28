import React from "react";
import { CourseProps } from "./Course.props";
import styles from "./Course.module.css";

const Course: React.FC<CourseProps> = ({ title, duration, description, url, imageSrc }) => {
  return (
    <a
      href={url}
      className={styles.Course}
      style={{ backgroundImage: `url(${imageSrc})` }} // Устанавливаем изображение как фон
    >
      <div className={styles.TextOverlay}>
        <p className={styles.Course_Duration}>{duration} hours</p>
        <p className={styles.Course_Description}>{description}</p>
        <h3 className={styles.Course_Title}>{title}</h3>
      </div>
    </a>
  );
};

export default Course;