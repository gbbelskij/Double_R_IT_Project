import { useEffect, useState } from "react";

import { FaCheck } from "react-icons/fa6";

import { CheckboxProps } from "./Checkbox.props";

import classes from "./Checkbox.module.css";
import classNames from "classnames";

const Checkbox: React.FC<CheckboxProps> = ({
  name,
  label,
  icon: Icon = FaCheck,
  register,
  error,
  defaultChecked = false,
}) => {
  const [isChecked, setIsChecked] = useState(defaultChecked);
  const { ref, onChange, ...rest } = register ? register(name) : {};

  useEffect(() => {
    setIsChecked(defaultChecked);
  }, [defaultChecked]);

  const handleChange = () => {
    const newValue = !isChecked;
    setIsChecked(newValue);

    if (onChange) {
      onChange({ target: { name, checked: newValue, type: "checkbox" } });
    }
  };

  return (
    <div className={classes.Checkbox}>
      <label className={classes.CheckboxWrapper} htmlFor={name}>
        <input
          className={classes.CheckboxInput}
          type="checkbox"
          id={name}
          name={name}
          checked={isChecked}
          onChange={handleChange}
          ref={ref}
          {...rest}
        />

        <span
          className={classNames(classes.Checkmark, {
            [classes.CheckmarkWithError]: error,
          })}
        >
          {isChecked && <Icon size={18} />}
        </span>

        {label}
      </label>

      {error && (
        <span className={classes.CheckboxErrorText}>{error.message}</span>
      )}
    </div>
  );
};

export default Checkbox;
