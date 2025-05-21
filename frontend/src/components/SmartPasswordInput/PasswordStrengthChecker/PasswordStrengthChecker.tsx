import { useEffect, useState } from "react";
import classNames from "classnames";

import { validatePasswordStrength } from "@utils/validatePasswordStrength";

import { PasswordStrengthCheckerProps } from "./PasswordStrengthChecker.props";

import classes from "./PasswordStrengthChecker.module.css";

const defaultText =
  "Убедитесь, что он содержит строчные, заглавные буквы латинского алфавита, цифры, спецсимволы и является достаточно длинным.";

const strengthLabels = [
  "Введите пароль",
  "Слабый пароль",
  "Средний пароль",
  "Хороший пароль",
];

const strengthColors = [
  "var(--solitude-100)",
  "var(--mandy-100)",
  "var(--chenin-100)",
  "var(--mantis-100)",
];

const PasswordStrengthChecker: React.FC<PasswordStrengthCheckerProps> = ({
  password,
  validateStrength = validatePasswordStrength,
  guidanceText = defaultText,
}) => {
  const [strength, setStrength] = useState(0);

  useEffect(() => {
    setStrength(validateStrength(password));
  }, [password, validateStrength]);

  return (
    <div
      className={classes.Checker}
      style={
        { "--strength-color": strengthColors[strength] } as React.CSSProperties
      }
      onMouseDown={(e) => e.preventDefault()}
    >
      <div className={classes.IndicatorWrapper}>
        {[1, 2, 3].map((level) => (
          <div
            key={level}
            className={classNames(classes.IndicatorBar, {
              [classes.IndicatorBarActive]: strength >= level,
            })}
          />
        ))}
      </div>

      <p className={classes.CheckerStrengthText}>
        {strengthLabels[validateStrength(password)]}
      </p>

      <p className={classes.CheckerGuidanceText}>{guidanceText}</p>
    </div>
  );
};

export default PasswordStrengthChecker;
