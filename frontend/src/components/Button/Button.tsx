import React from "react";
import classNames from "classnames";

import { ButtonProps } from "./Button.props";

import classes from "./Button.module.css";

const Button: React.FC<ButtonProps> = ({
  children,
  type = "button",
  size = "small",
  color = "default",
  isFullWidth = false,
  leftIcon,
  rightIcon,
  onClick = () => {},
  ...props
}) => {
  const buttonClasses = classNames(
    classes.Button,
    {
      [classes.ButtonDefault]: color === "default",
      [classes.ButtonInverse]: color === "inverse",
      [classes.ButtonDim]: color === "dim",
      [classes.ButtonGreen]: color === "green",
      [classes.ButtonRed]: color === "red",

      [classes.ButtonSmall]: size === "small",
      [classes.ButtonMedium]: size === "medium",
      [classes.ButtonLarge]: size === "large",

      [classes.ButtonFullWidth]: isFullWidth,
    },
    props.className
  );

  return (
    <button {...props} className={buttonClasses} type={type} onClick={onClick}>
      {leftIcon}
      <span className={classes.ButtonContent}>{children}</span>
      {rightIcon}
    </button>
  );
};

export default Button;
