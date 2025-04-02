import React from "react";
import classes from "./CheckboxButtonGroup.module.css";
import CheckboxButton from "../CheckboxButton/CheckboxButton";
import { CheckboxButtonGroupProps } from "./CheckboxButtonGroup.props";

export const CheckboxButtonGroup: React.FC<CheckboxButtonGroupProps> = ({
  options,
  selectedOptions,
  onChange,
}) => {
  const handleClick = (value: string) => {
    if (selectedOptions.includes(value)) {
      onChange(selectedOptions.filter((option) => option !== value));
    } else {
      onChange([...selectedOptions, value]);
    }
  };

  return (
    <div className={classes.groupContainer}>
      {options.map((option) => (
        <CheckboxButton
          key={option}
          value={option}
          isSelected={selectedOptions.includes(option)}
          onClick={handleClick}
        />
      ))}
    </div>
  );
};