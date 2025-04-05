import { useEffect, useRef, useState } from "react";
import shave from "shave";

import { CourseProps } from "./Course.props";

import classes from "./Course.module.css";

const declineMonth = (duration: number): string => {
  const lastDigit = duration % 10;
  const lastTwoDigits = duration % 100;

  if (lastTwoDigits >= 11 && lastTwoDigits <= 14) {
    return `${duration} месяцев`;
  }

  if (lastDigit === 1) {
    return `${duration} месяц`;
  }

  if (lastDigit >= 2 && lastDigit <= 4) {
    return `${duration} месяца`;
  }

  return `${duration} месяцев`;
};

const HOVERED_DESCRIPTION_HEIGHT = 400;
const UNHOVERED_DESCRIPTION_HEIGHT = 44;

const Course: React.FC<CourseProps> = ({
  title,
  duration,
  description,
  url,
  imageSrc,
}) => {
  const descriptionRef = useRef(null);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const descriptionElement = descriptionRef.current;

    if (!descriptionElement) return;

    const handleResize = () => {
      setTimeout(() => {
        shave(
          descriptionElement,
          isHovered ? HOVERED_DESCRIPTION_HEIGHT : UNHOVERED_DESCRIPTION_HEIGHT
        );
      }, 100);
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [isHovered]);

  return (
    <a
      href={url}
      className={classes.Course}
      target="_blank"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className={classes.CourseOverlay} />

      <div className={classes.CourseInfoSection}>
        <p className={classes.CourseDuration}>
          Курс · {declineMonth(duration)}
        </p>
        <h3 className={classes.CourseTitle}>{title}</h3>
        <p className={classes.CourseDescription} ref={descriptionRef}>
          {description}
        </p>
      </div>

      <div
        className={classes.CourseBackground}
        style={{ backgroundImage: `url(${imageSrc})` }}
      />
    </a>
  );
};

export default Course;
