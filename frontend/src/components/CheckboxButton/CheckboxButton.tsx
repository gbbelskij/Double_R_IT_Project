import { CheckboxButtonProps } from "./CheckboxButton.props";

import classes from "./CheckboxButton.module.css";

const CheckboxButton: React.FC<CheckboxButtonProps> = ({
  value,
  isSelected,
  onClick,
}) => {
  return (
    <div className={classes.checkboxContainer}>
      <input
        type="checkbox"
        id={value}
        checked={isSelected}
        onChange={() => onClick(value)}
        className={classes.input}
      />
      <label htmlFor={value} className={classes.label}>
        {value}
      </label>
    </div>
  );
};

export default CheckboxButton;
