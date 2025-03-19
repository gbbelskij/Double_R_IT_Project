import React from "react";
import classNames from "classnames";

import { ButtonProps } from "./Button.props";
import { ButtonTypes } from "./enums/ButtonTypes";

import classes from "./Button.module.css";

const Button: React.FC<ButtonProps> = ({
  type,
  fullWidth = false,
  leftIcon,
  rightIcon,
  children,
}) => {
  const buttonClasses = classNames(classes.Button, {
    [classes.ButtonAction]: type === ButtonTypes.Action,
    [classes.ButtonDangerousAction]: type === ButtonTypes.DangerousAction,
    [classes.ButtonDefault]: type === ButtonTypes.Default,
    [classes.ButtonSubmit]: type === ButtonTypes.Submit,
    [classes.ButtonFullWidth]: fullWidth,
  });

  return (
    <button
      className={buttonClasses}
      type={type === ButtonTypes.Submit ? "submit" : "button"}
    >
      {leftIcon && <span className={classes.ButtonIconLeft}>{leftIcon}</span>}
      <span className={classes.ButtonContent}>{children}</span>
      {rightIcon && (
        <span className={classes.ButtonIconRight}>{rightIcon}</span>
      )}
    </button>
  );
};

export default Button;
