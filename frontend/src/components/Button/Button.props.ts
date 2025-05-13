import { ReactNode } from "react";
import { ButtonColors, ButtonSizes, ButtonTypes } from "./Button.types";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /**
   * Content inside the button (usually text)
   */
  children: ReactNode;
  /**
   * Button type
   */
  type?: ButtonTypes;
  /**
   * Button size
   */
  size?: ButtonSizes;
  /**
   * Button color scheme
   */
  color?: ButtonColors;
  /**
   * Whether the button should take full width of its container
   */
  isFullWidth?: boolean;
  /**
   * Icon displayed on the left side of the button
   */
  leftIcon?: ReactNode;
  /**
   * Icon displayed on the right side of the button
   */
  rightIcon?: ReactNode;
  /**
   * Click handler
   */
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
}
