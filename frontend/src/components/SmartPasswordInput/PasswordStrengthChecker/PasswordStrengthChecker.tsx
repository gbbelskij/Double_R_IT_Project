import { useEffect, useState } from "react";
import classNames from "classnames";

import { PasswordStrengthCheckerProps } from "./PasswordStrengthChecker.props";

import classes from "./PasswordStrengthChecker.module.css";

const defaultText =
  "Убедитесь, что он содержит строчные, заглавные буквы латинского алфавита, цифры, спецсимволы и является достаточно длинным.";

const defaultValidateStrength = (password: string): number => {
  if (password.length === 0) return 0;

  const hasLowercase = /[a-z]/.test(password);
  const hasUppercase = /[A-Z]/.test(password);
  const hasDigit = /\d/.test(password);
  const hasSpecialChar = /[^\w\s]/.test(password);
  const hasValidLength = password.length >= 8;

  if (!hasValidLength) return 1;

  let score = 0;

  if (password.length >= 12) score += 2;
  else if (password.length >= 8) score += 1;

  const diversityFactors = [
    hasLowercase,
    hasUppercase,
    hasDigit,
    hasSpecialChar,
    password.length > 14,
  ].filter(Boolean).length;

  score += diversityFactors;

  if (score <= 3) return 1;
  if (score <= 5) return 2;
  return 3;
};

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
  validateStrength = defaultValidateStrength,
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
