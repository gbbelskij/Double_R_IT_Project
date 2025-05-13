import { RadioButtonProps } from "./RadioButton.props";

import classes from "./RadioButton.module.css";

const RadioButton: React.FC<RadioButtonProps> = ({
  value,
  displayedValue,
  isSelected,
  onClick,
}) => {
  return (
    <div className={classes.Radio}>
      <input
        type="radio"
        id={`${value}-${displayedValue}`}
        checked={isSelected}
        onChange={() => onClick(value)}
        className={classes.RadioInput}
      />
      <label
        htmlFor={`${value}-${displayedValue}`}
        className={classes.RadioLabel}
      >
        {displayedValue}
      </label>
    </div>
  );
};

export default RadioButton;
