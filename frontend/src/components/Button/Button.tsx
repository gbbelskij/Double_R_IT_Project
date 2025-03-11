import React from 'react';
import './Button.css'; // Ensure this path is correct

// Export ButtonTypes enum
export enum ButtonTypes {
  Action = 'action',
  DangerousAction = 'dangerous-action',
  Default = 'default',
  Submit = 'submit',
}

// Define props interface
interface ButtonProps {
  type: ButtonTypes;
  fullWidth?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  children: React.ReactNode;
}

// Button component
const Button: React.FC<ButtonProps> = ({
  type,
  fullWidth = false,
  leftIcon,
  rightIcon,
  children,
}) => {
  return (
    <button
      className={`button ${type} ${fullWidth ? 'full-width' : ''}`}
      type={type === ButtonTypes.Submit ? 'submit' : 'button'}
    >
      {leftIcon && <span className="icon left">{leftIcon}</span>}
      <span className="content">{children}</span>
      {rightIcon && <span className="icon right">{rightIcon}</span>}
    </button>
  );
};

export default Button;