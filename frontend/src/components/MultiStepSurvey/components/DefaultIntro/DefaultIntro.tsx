import Button from "@components/Button/Button";

import { IntroProps } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import classes from "./DefaultIntro.module.css";

const DefaultIntro: React.FC<IntroProps> = ({ onStepChange }) => {
  return (
    <div className={classes.DefaultIntro}>
      <div className={classes.DefaultIntroSection}>
        <h2 className={classes.DefaultIntroHeading}>Регистрация</h2>

        <p>
          Поздравляем! Вы сделали первый шаг к новым карьерным возможностям!
        </p>

        <p>
          Теперь вам предстоит пройти небольшой тест по профориентации. Он
          поможет определить ваши сильные стороны и подобрать наиболее
          подходящие курсы для успешного перехода в новую профессию.
        </p>
      </div>

      <Button
        size="large"
        color="inverse"
        isFullWidth
        onClick={() => onStepChange(0)}
      >
        Приступить к тесту
      </Button>
    </div>
  );
};

export default DefaultIntro;
