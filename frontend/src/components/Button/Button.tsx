import React from 'react';
import classNames from 'classnames';
import { ButtonProps } from './Button/Button.props';
import { ButtonTypes } from './enums/ButtonTypes';
import styles from './Button.module.css';

const Button: React.FC<ButtonProps> = ({
  type,
  fullWidth = false,
  leftIcon,
  rightIcon,
  children,
}) => {
  const buttonClasses = classNames(styles.Button, {
    [styles.ButtonAction]: type === ButtonTypes.Action,
    [styles.ButtonDangerousAction]: type === ButtonTypes.DangerousAction,
    [styles.ButtonDefault]: type === ButtonTypes.Default,
    [styles.ButtonSubmit]: type === ButtonTypes.Submit,
    [styles.ButtonFullWidth]: fullWidth,
  });

  return (
    <button
      className={buttonClasses}
      type={type === ButtonTypes.Submit ? 'submit' : 'button'}
    >
      {leftIcon && <span className={styles.ButtonIconLeft}>{leftIcon}</span>}
      <span className={styles.ButtonContent}>{children}</span>
      {rightIcon && <span className={styles.ButtonIconRight}>{rightIcon}</span>}
    </button>
  );
};

export default Button;