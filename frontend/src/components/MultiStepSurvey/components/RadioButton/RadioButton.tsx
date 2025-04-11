import { extractAnswer } from "@components/MultiStepSurvey/utils";

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
        id={extractAnswer(value)}
        checked={isSelected}
        onChange={() => onClick(value)}
        className={classes.RadioInput}
      />
      <label htmlFor={extractAnswer(value)} className={classes.RadioLabel}>
        {extractAnswer(value)}
      </label>
    </div>
  );
};

export default RadioButton;
