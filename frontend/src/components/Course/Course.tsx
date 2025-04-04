import React from "react";
import { CourseProps } from "./Course.props";
import styles from "./Course.module.css";

const Course: React.FC<CourseProps> = ({
  title,
  duration,
  description,
  url,
  imageSrc,
}) => {
  return (
    <a href={url} className={styles.Course}>
      {/* Фоновое изображение */}
      <div
        className={styles.CourseBackground}
        style={{ backgroundImage: `url(${imageSrc})` }}
      />

      {/* Затемняющий слой */}
      <div className={styles.CourseOverlay} />

      {/* Текст поверх */}
      <div className={styles.TextOverlay}>
        <p className={styles.Course_Duration}>Курс · {duration} месяцев</p>
        <h3 className={styles.Course_Title}>{title}</h3>
        <p className={styles.Course_Description}>{description}</p>
      </div>
    </a>
  );
};

export default Course;
