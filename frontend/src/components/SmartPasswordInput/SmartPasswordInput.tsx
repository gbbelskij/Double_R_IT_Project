import { useRef, useState } from "react";
import classNames from "classnames";

import { LuKeyRound } from "react-icons/lu";
import { LuEye } from "react-icons/lu";
import { LuEyeOff } from "react-icons/lu";
import { TbRepeat } from "react-icons/tb";

import { useWindowSize } from "@hooks/useWindowSize";

import Input from "@components/Input/Input";
import PasswordStrengthChecker from "./PasswordStrengthChecker/PasswordStrengthChecker";

import { SmartPasswordInputProps } from "./SmartPasswordInput.props";

import classes from "./SmartPasswordInput.module.css";

const SmartPasswordInput: React.FC<SmartPasswordInputProps> = ({
  name = "password",
  label = "Пароль",
  placeholder,
  icon,
  register,
  error,
  repeatName = "repeatPassword",
  repeatLabel = "Повторите пароль",
  repeatPlaceholder,
  repeatIcon,
  repeatError,
}) => {
  const [password, setPassword] = useState<string>("");
  const [isPasswordVisible, setIsPasswordVisible] = useState(false);
  const [isFocused, setIsFocused] = useState(false);

  const { isSmallMobile } = useWindowSize();

  const wrapperRef = useRef<HTMLDivElement>(null);
  const handleFocus = () => setIsFocused(true);
  const handleBlur = (e: React.FocusEvent) => {
    if (
      wrapperRef.current &&
      e.relatedTarget &&
      wrapperRef.current.contains(e.relatedTarget as Node)
    ) {
      return;
    }

    setIsFocused(false);
  };

  const inputType = isPasswordVisible ? "text" : "password";
  const iconSize = isSmallMobile ? 20 : 28;

  const togglePasswordVisibility = () => setIsPasswordVisible((prev) => !prev);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPassword(e.target.value);
  };

  const PasswordInputIcon = icon || LuKeyRound;
  const PasswordRepeatInputIcon = repeatIcon || TbRepeat;
  const PasswordVisibilityIcon = isPasswordVisible ? LuEye : LuEyeOff;

  return (
    <>
      <div
        ref={wrapperRef}
        onFocusCapture={handleFocus}
        onBlurCapture={handleBlur}
        className={classes.Wrapper}
      >
        <label className={classes.PasswordInputField} htmlFor={name}>
          <div className={classes.PasswordInputLabel}>{label}</div>

          <div
            className={classNames(classes.PasswordInputWrapper, {
              [classes.PasswordInputWrapperWithError]: error,
            })}
          >
            <PasswordInputIcon size={iconSize} />

            <input
              className={classes.PasswordInput}
              id={name}
              type={inputType}
              placeholder={placeholder}
              {...(register ? register(name) : {})}
              onChange={(e) => {
                handleChange(e);

                if (register) {
                  register(name).onChange(e);
                }
              }}
            />

            <PasswordVisibilityIcon
              size={iconSize}
              onMouseDown={(e) => e.preventDefault()}
              onClick={togglePasswordVisibility}
              className={classes.PasswordInputButton}
            />
          </div>

          {error && (
            <p className={classes.PasswordInputErrorText}>{error.message}</p>
          )}
        </label>

        {isFocused && <PasswordStrengthChecker password={password} />}
      </div>

      {!isFocused && password && (
        <Input
          type="password"
          name={repeatName}
          placeholder={repeatPlaceholder}
          label={repeatLabel}
          icon={PasswordRepeatInputIcon}
          register={register}
          error={repeatError}
        />
      )}
    </>
  );
};

export default SmartPasswordInput;
