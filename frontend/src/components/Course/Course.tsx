import { useLayoutEffect, useRef, useState } from "react";
import shave from "shave";

import useWindowWidth from "@hooks/useWindowWidth";
import { declineMonth } from "@utils/decline";

import { CourseProps } from "./Course.props";

import classes from "./Course.module.css";

const Course: React.FC<CourseProps> = ({
  title,
  duration,
  description,
  url,
  imageSrc,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const descriptionRef = useRef<HTMLParagraphElement>(null);

  const windowWidth = useWindowWidth();

  const HOVERED_DESCRIPTION_HEIGHT = windowWidth <= 1700 ? 130 : 200;
  const DEFAULT_DESCRIPTION_HEIGHT = windowWidth <= 375 ? 76 : 150;
  const UNHOVERED_DESCRIPTION_HEIGHT = 44;

  useLayoutEffect(() => {
    const descriptionElement = descriptionRef.current;

    if (!descriptionElement) {
      return;
    }

    const height =
      windowWidth >= 768
        ? isHovered
          ? HOVERED_DESCRIPTION_HEIGHT
          : UNHOVERED_DESCRIPTION_HEIGHT
        : DEFAULT_DESCRIPTION_HEIGHT;

    if (!isHovered) {
      shave(descriptionElement, height);

      return;
    }

    const timer = setTimeout(() => {
      shave(descriptionElement, height);
    }, 100);

    return () => clearTimeout(timer);
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
