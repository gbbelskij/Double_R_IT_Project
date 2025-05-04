import { useState } from "react";
import { InputMask } from "@react-input/mask";
import { IconType } from "react-icons";
import classNames from "classnames";

import { FaCheck } from "react-icons/fa6";
import { FaRegUserCircle } from "react-icons/fa";
import { TbCalendarQuestion } from "react-icons/tb";
import { AiOutlineMail } from "react-icons/ai";
import { MdAccessTime } from "react-icons/md";
import { LuKeyRound } from "react-icons/lu";
import { LuEye } from "react-icons/lu";
import { LuEyeOff } from "react-icons/lu";

import { useWindowSize } from "@hooks/useWindowSize";

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
  register,
  error,
  valueAsNumber,
}) => {
  const [value, setValue] = useState<string>(defaultValue);
  const [width, setWidth] = useState<string>("18px");
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);

  const { isSmallMobile } = useWindowSize();

  const inputType =
    type === "password" ? (isPasswordVisible ? "text" : "password") : type;
  const iconSize = isSmallMobile ? 20 : 28;

  const togglePasswordVisibility = () => setIsPasswordVisible((prev) => !prev);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setValue(e.target.value);
  };

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;

    const parsed = Number(newValue);

    if (!isNaN(parsed)) {
      setValue(newValue);

      if (parsed >= 0 && parsed <= 9) {
        setWidth("18px");
      } else {
        setWidth("33px");
      }
    }
  };

  const isValidNumber = !isNaN(Number(value)) && value !== "";

  const registered = register
    ? register(
        name,
        valueAsNumber
          ? {
              valueAsNumber: true,
              setValueAs: (v) => (isNaN(v) ? 0 : Number(v)),
            }
          : {}
      )
    : undefined;

  const InputIcon = icon || getDefaultIcon(type);
  const PasswordVisibilityIcon = isPasswordVisible ? LuEye : LuEyeOff;

  return (
    <label className={classes.InputField} htmlFor={name}>
      <div className={classes.InputLabel}>{label}</div>

      <div
        className={classNames(classes.InputWrapper, {
          [classes.InputWrapperWithError]: error,
        })}
      >
        {InputIcon && <InputIcon size={iconSize} />}

        {type === "experience" ? (
          <InputMask
            style={{ width: width }}
            className={classNames(classes.Input, classes.MaskedInput)}
            id={name}
            mask="__"
            replacement={{ _: /\d/ }}
            {...registered}
            onChange={(e) => {
              handleNumberChange(e);
              registered?.onChange(e);
            }}
            placeholder={placeholder}
          />
        ) : (
          <input
            className={classes.Input}
            id={name}
            type={inputType}
            placeholder={
              placeholder
                ? placeholder
                : type === "email"
                  ? "example@mail.com"
                  : undefined
            }
            {...registered}
            onChange={(e) => {
              handleChange(e);
              registered?.onChange(e);
            }}
          />
        )}

        {type === "experience" && getUnit && isValidNumber && (
          <span className={classes.UnitText}>{getUnit(parseInt(value))}</span>
        )}

        {type === "password" && (
          <PasswordVisibilityIcon
            size={iconSize}
            onClick={togglePasswordVisibility}
            className={classes.InputButton}
          />
        )}
      </div>

      {error && <p className={classes.InputErrorText}>{error.message}</p>}
    </label>
  );
};

export default Input;
