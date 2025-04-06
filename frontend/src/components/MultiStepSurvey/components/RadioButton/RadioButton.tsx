import { RadioButtonProps } from "./RadioButton.props";

import classes from "./RadioButton.module.css";

const RadioButton: React.FC<RadioButtonProps> = ({
  value,
  isSelected,
  onClick,
}) => {
  return (
    <div className={classes.Radio}>
      <input
        type="radio"
        id={value}
        checked={isSelected}
        onChange={() => onClick(value)}
        className={classes.RadioInput}
      />
      <label htmlFor={value} className={classes.RadioLabel}>
        {value}
      </label>
    </div>
  );
};

export default RadioButton;
