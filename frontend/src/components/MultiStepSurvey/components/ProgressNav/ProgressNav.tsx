import classNames from "classnames";

import { ProgressNavProps } from "./ProgressNav.props";

import classes from "./ProgressNav.module.css";

const ProgressNav: React.FC<ProgressNavProps> = ({
  selectedAnswers,
  onStepChange,
}) => {
  return (
    <nav className={classes.ProgressNav}>
      {selectedAnswers.map((item, index) => (
        <div
          className={classNames(classes.ProgressNavItem, {
            [classes.ProgressNavItemActive]: item.answer,
          })}
          key={index}
          onClick={() => onStepChange(index)}
          title={item.question}
        />
      ))}
    </nav>
  );
};

export default ProgressNav;
