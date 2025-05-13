import classNames from "classnames";

import { ProgressNavProps } from "./ProgressNav.props";

import classes from "./ProgressNav.module.css";

const ProgressNav: React.FC<ProgressNavProps> = ({
  questions,
  questionIDs,
  surveyData,
  onStepChange,
  step,
}) => {
  return (
    <nav className={classes.ProgressNav}>
      {questionIDs.map((item, index) => (
        <div
          className={classNames(classes.ProgressNavItem, {
            [classes.ProgressNavItemActive]: surveyData[item] !== null,
            [classes.ProgressNavItemCurrent]: step === index,
          })}
          key={index}
          title={questions[item].text}
          onClick={() => onStepChange(index)}
        />
      ))}
    </nav>
  );
};

export default ProgressNav;
