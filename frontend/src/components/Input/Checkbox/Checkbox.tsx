import { useState } from "react";
import { FaCheck } from "react-icons/fa6";

import { CheckboxProps } from "./Checkbox.props";

import classes from "./Checkbox.module.css";

const Checkbox: React.FC<CheckboxProps> = ({
  name,
  label,
  icon: Icon = FaCheck,
}) => {
  const [isChecked, setIsChecked] = useState(false);

  return (
    <label className={classes.Checkbox} htmlFor={name}>
      <input
        className={classes.CheckboxInput}
        type="checkbox"
        id={name}
        name={name}
        checked={isChecked}
        onChange={() => setIsChecked((prev) => !prev)}
      />

      <span className={classes.Checkmark}>
        {isChecked && <Icon size={18} />}
      </span>

      {label}
    </label>
  );
};

export default Checkbox;
