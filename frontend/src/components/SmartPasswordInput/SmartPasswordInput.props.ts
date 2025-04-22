import { IconType } from "react-icons";
import { FieldError, UseFormRegister } from "react-hook-form";

export interface SmartPasswordInputProps {
  /**
   * Name attribute of the password input element
   */
  name?: string;
  /**
   * Optional label displayed above or beside the password input
   */
  label?: string;
  /**
   * Optional icon to display inside the password input field
   */
  icon?: IconType;
  /**
   * Placeholder text shown when password input is empty
   */
  placeholder?: string;
  /**
   * Function to register the password input field with react-hook-form.
   * Enables form validation, value tracking, and event handling.
   */
  register?: UseFormRegister<any>;
  /**
   * Validation error associated with the password input field.
   * Used to display error messages and apply error styling.
   */
  error?: FieldError;
  /**
   * Name attribute of the password repeat input element
   */
  repeatName?: string;
  /**
   * Optional label displayed above or beside the password repeat input
   */
  repeatLabel?: string;
  /**
   * Optional icon to display inside the password repeat input field
   */
  repeatIcon?: IconType;
  /**
   * Placeholder text shown when password repeat input is empty
   */
  repeatPlaceholder?: string;
  /**
   * Validation error associated with the password repeat input field.
   * Used to display error messages and apply error styling.
   */
  repeatError?: FieldError;
}
