import { IconType } from "react-icons";
import { SelectOption } from "./Select.types";
import { FieldError, UseFormRegister } from "react-hook-form";

export interface SelectProps {
  /**
   * Options of select
   */
  options: SelectOption[];
  /**
   * Name attribute of the select element
   */
  name: string;
  /**
   * Optional label displayed above or beside the select
   */
  label?: string;
  /**
   * Optional icon to display inside the select field
   */
  icon?: IconType;
  /**
   * Default value of the select, must be in options
   */
  defaultValue?: string;
  /**
   * Placeholder text shown when select is empty
   */
  placeholder?: string;
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
   * @todo
   */
  watch?: (name: string) => string;
}
