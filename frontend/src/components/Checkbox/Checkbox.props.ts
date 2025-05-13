import { FieldError, UseFormRegister } from "react-hook-form";

import { IconType } from "react-icons";

export interface CheckboxProps {
  /**
   * Name of the checkbox input.
   */
  name: string;
  /**
   * Optional label to display next to the checkbox.
   */
  label?: string;
  /**
   * Optional icon to display with the checkbox.
   */
  icon?: IconType;
  /**
   * Function to register the input field with react-hook-form.
   * Enables form validation, value tracking, and event handling.
   */
  register?: UseFormRegister<any>;
  /**
   * Validation error associated with the input field.
   * Used to display error messages and apply error styling.
   */
  error?: FieldError;
  /**
   * The flag indicates whether the checkbox is checked.
   */
  defaultChecked?: boolean;
}
