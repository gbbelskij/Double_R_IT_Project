import classNames from "classnames";

import { ProgressNavProps } from "./ProgressNav.props";

import classes from "./ProgressNav.module.css";

const ProgressNav: React.FC<ProgressNavProps> = ({
  questions,
  answerIDs,
  selectedAnswers,
  onStepChange,
}) => {
  return (
    <nav className={classes.ProgressNav}>
      {answerIDs.map((item, index) => (
        <div
          className={classNames(classes.ProgressNavItem, {
            [classes.ProgressNavItemActive]: selectedAnswers[item],
          })}
          key={index}
          onClick={() => onStepChange(index)}
          title={questions[item].text}
        />
      ))}
    </nav>
  );
};

export default ProgressNav;
