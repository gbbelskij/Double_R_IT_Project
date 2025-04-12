import classNames from "classnames";
import { useState } from "react";
import { InputMask } from "@react-input/mask";

import { IconType } from "react-icons";

import { FaCheck } from "react-icons/fa6";
import { FaRegUserCircle } from "react-icons/fa";
import { TbCalendarQuestion } from "react-icons/tb";
import { AiOutlineMail } from "react-icons/ai";
import { MdAccessTime } from "react-icons/md";
import { LuKeyRound } from "react-icons/lu";
import { LuEye } from "react-icons/lu";
import { LuEyeOff } from "react-icons/lu";

import { InputTypes } from "./Input.types";
import { InputProps } from "./Input.props";

import classes from "./Input.module.css";

const defaultIcons: Record<string, IconType> = {
  text: FaRegUserCircle,
  date: TbCalendarQuestion,
  email: AiOutlineMail,
  experience: MdAccessTime,
  password: LuKeyRound,
  checkbox: FaCheck,
};

const getDefaultIcon = (type: InputTypes): IconType | null => {
  return defaultIcons[type] || null;
};

const Input: React.FC<InputProps> = ({
  type,
  name,
  label,
  placeholder,
  icon,
  defaultValue = "",
  getUnit,
}) => {
  const [value, setValue] = useState<string>(defaultValue);
  const [width, setWidth] = useState<string>("16px");
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);

  const inputType =
    type === "password" ? (isPasswordVisible ? "text" : "password") : type;

  const togglePasswordVisibility = () => setIsPasswordVisible((prev) => !prev);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setValue(e.target.value);
  };

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;

    setValue(newValue);

    if (0 <= parseInt(newValue) && parseInt(newValue) <= 9) {
      setWidth("16px");
    } else {
      setWidth("33px");
    }
  };

  const InputIcon = icon || getDefaultIcon(type);
  const PasswordVisibilityIcon = isPasswordVisible ? LuEye : LuEyeOff;

  return (
    <label className={classes.InputField} htmlFor={name}>
      <div className={classes.InputLabel}>{label}</div>

      <div className={classes.InputWrapper}>
        {InputIcon && <InputIcon size={28} />}

        {type === "experience" ? (
          <InputMask
            style={{ width: width }}
            className={classNames(classes.Input, classes.MaskedInput)}
            id={name}
            name={name}
            mask="__"
            replacement={{ _: /\d/ }}
            onChange={handleNumberChange}
            placeholder={placeholder}
          />
        ) : (
          <input
            className={classes.Input}
            id={name}
            name={name}
            type={inputType}
            onChange={handleChange}
            placeholder={
              placeholder
                ? placeholder
                : type === "email"
                  ? "example@mail.com"
                  : undefined
            }
          />
        )}

        {type === "experience" && getUnit && value.trim() !== "" && (
          <span className={classes.UnitText}>{getUnit(parseInt(value))}</span>
        )}

        {type === "password" && (
          <PasswordVisibilityIcon
            size={28}
            onClick={togglePasswordVisibility}
            className={classes.InputButton}
          />
        )}
      </div>
    </label>
  );
};

export default Input;
