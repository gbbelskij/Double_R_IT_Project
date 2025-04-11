import { useEffect, useMemo, useRef, useState } from "react";
import shave from "shave";

import useWindowWidth from "@hooks/useWindowWidth";

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

const Course: React.FC<CourseProps> = ({
  title,
  duration,
  description,
  url,
  imageSrc,
}) => {
  const descriptionRef = useRef<HTMLParagraphElement>(null);
  const [isHovered, setIsHovered] = useState(false);
  const windowWidth = useWindowWidth();

  const HOVERED_DESCRIPTION_HEIGHT = useMemo(
    () => (windowWidth <= 1700 ? 130 : 200),
    [windowWidth]
  );
  const DEFAULT_DESCRIPTION_HEIGHT = useMemo(
    () => (windowWidth <= 375 ? 76 : 150),
    [windowWidth]
  );
  const UNHOVERED_DESCRIPTION_HEIGHT = 44;

  useEffect(() => {
    const descriptionElement = descriptionRef.current;

    if (descriptionElement) {
      if (windowWidth >= 768) {
        setTimeout(() => {
          shave(
            descriptionElement,
            isHovered
              ? HOVERED_DESCRIPTION_HEIGHT
              : UNHOVERED_DESCRIPTION_HEIGHT
          );
        }, 100);
      } else {
        shave(descriptionElement, DEFAULT_DESCRIPTION_HEIGHT);
      }
    }
  }, [
    isHovered,
    windowWidth,
    description,
    HOVERED_DESCRIPTION_HEIGHT,
    DEFAULT_DESCRIPTION_HEIGHT,
  ]);

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
