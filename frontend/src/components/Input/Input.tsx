import { useState } from "react";
import { InputMask } from "@react-input/mask";

import { InputTypes } from "./Input.types";
import { InputProps } from "./Input.props";

import Checkbox from "./Checkbox/Checkbox";

import { FaCheck } from "react-icons/fa6";
import { FaRegUserCircle } from "react-icons/fa";
import { TbCalendarQuestion } from "react-icons/tb";
import { AiOutlineMail } from "react-icons/ai";
import { MdAccessTime } from "react-icons/md";
import { LuKeyRound } from "react-icons/lu";
import { LuEye } from "react-icons/lu";
import { LuEyeOff } from "react-icons/lu";

import classes from "./Input.module.css";
import classNames from "classnames";

const defaultIcons = {
  text: FaRegUserCircle,
  date: TbCalendarQuestion,
  email: AiOutlineMail,
  number: MdAccessTime,
  password: LuKeyRound,
  checkbox: FaCheck,
};

const getDefaultIcon = (type: InputTypes) => {
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
  const [value, setValue] = useState(defaultValue);
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);

  const inputType =
    type === "password" ? (isPasswordVisible ? "text" : "password") : type;

  const togglePasswordVisibility = () => setIsPasswordVisible((prev) => !prev);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setValue(e.target.value);
  };

  const InputIcon = icon || getDefaultIcon(type);
  const PasswordVisibilityIcon = isPasswordVisible ? LuEye : LuEyeOff;

  if (type === "checkbox") {
    return <Checkbox name={name} label={label} />;
  }

  return (
    <label className={classes.InputField} htmlFor={name}>
      <div className={classes.InputLabel}>{label}</div>

      <div className={classes.InputWrapper}>
        {InputIcon && <InputIcon size={28} />}

        {type === "number" ? (
          <InputMask
            className={classNames(classes.Input, classes.MaskedInput)}
            id={name}
            name={name}
            mask="__"
            replacement={{ _: /\d/ }}
            value={value}
            onChange={handleChange}
            placeholder={placeholder}
          />
        ) : (
          <input
            className={classes.Input}
            id={name}
            name={name}
            type={inputType}
            value={value}
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

        {type === "number" && getUnit && value.trim() !== "" && (
          <span className={classes.UnitText}>{getUnit(parseInt(value))}</span>
        )}

        {type === "password" && (
          <PasswordVisibilityIcon
            size={28}
            onClick={togglePasswordVisibility}
          />
        )}
      </div>
    </label>
  );
};

export default Input;
