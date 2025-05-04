import { IconType } from "react-icons";
import { InputTypes } from "./Input.types";
import { FieldError, UseFormRegister } from "react-hook-form";

export interface InputProps {
  /**
   * Type of the input (e.g., text, number, password)
   */
  type: InputTypes;
  /**
   * Name attribute of the input element
   */
  name: string;
  /**
   * Optional label displayed above or beside the input
   */
  label?: string;
  /**
   * Optional icon to display inside the input field
   */
  icon?: IconType;
  /**
   * Default value of the input
   */
  defaultValue?: string;
  /**
   * Placeholder text shown when input is empty
   */
  placeholder?: string;
  /**
   * Optional function that returns a unit string based on the numeric input value
   */
  getUnit?: (value: number) => string;
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
   * A flag indicating that the value will be saved as a number.
   */
  valueAsNumber?: boolean;
}
