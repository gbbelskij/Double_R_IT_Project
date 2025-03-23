import { IconType } from "react-icons";
import { InputTypes } from "./Input.types";

export interface InputProps {
  type: InputTypes;
  name: string;
  label?: string;
  icon?: IconType;
  defaultValue?: string;
  placeholder?: string;
  getUnit?: (value: number) => string;
}
