import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import shave from "shave";

import { useWindowSize } from "@hooks/useWindowSize";

import { declineMonth } from "@utils/decline";
import { getDominantColorFromImage } from "@utils/getDominantColorFromImage";

import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import { CourseProps } from "./Course.props";

import classes from "./Course.module.css";
import classNames from "classnames";

const Course: React.FC<CourseProps> = ({
  title,
  duration,
  description,
  url,
  imageUrl = "assets/img/course_default_picture.png",
  colorOptions,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [color, setColor] = useState<string>("");

  const descriptionRef = useRef<HTMLParagraphElement>(null);
  const cardRef = useRef(null);

  const { width: windowWidth, isSmallMobile, isMobile } = useWindowSize();

  const HOVERED_DESCRIPTION_HEIGHT = useMemo(
    () => (windowWidth <= 1700 ? 130 : 200),
    [windowWidth]
  );

  const DEFAULT_DESCRIPTION_HEIGHT = useMemo(
    () => (isSmallMobile ? 90 : 150),
    [isSmallMobile]
  );

  const UNHOVERED_DESCRIPTION_HEIGHT = 44;

  useLayoutEffect(() => {
    const descriptionElement = descriptionRef.current;

    if (!descriptionElement) {
      return;
    }

    const height = isMobile
      ? DEFAULT_DESCRIPTION_HEIGHT
      : isHovered
        ? HOVERED_DESCRIPTION_HEIGHT
        : UNHOVERED_DESCRIPTION_HEIGHT;

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
    isMobile,
    description,
    DEFAULT_DESCRIPTION_HEIGHT,
    HOVERED_DESCRIPTION_HEIGHT,
  ]);

  useEffect(() => {
    getDominantColorFromImage(imageUrl, colorOptions)
      .then((color) => {
        setColor(color);
      })
      .catch((error) => {
        console.error("Failed to get color", error);
      });
  }, [imageUrl, colorOptions]);

  return (
    <a
      href={url}
      className={classes.Course}
      target="_blank"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      ref={cardRef}
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
        style={{ backgroundImage: `url(${imageUrl})` }}
      />

      <BackgroundElements
        targetRef={cardRef}
        count={1}
        color={color}
        blobsSize={isSmallMobile ? 250 : 400}
        styles={classNames(classes.CourseBackgroundElements, {
          [classes.HoveredCourseBackgroundElements]: isHovered,
        })}
      />
    </a>
  );
};

export default Course;
