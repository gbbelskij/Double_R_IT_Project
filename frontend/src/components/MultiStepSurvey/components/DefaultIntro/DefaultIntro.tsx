import useWindowWidth from "@hooks/useWindowWidth";

import Button from "@components/Button/Button";
import LogoContainer from "@components/LogoContainer/LogoContainer";

import { IntroProps } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import classes from "./DefaultIntro.module.css";

const DefaultIntro: React.FC<IntroProps> = ({ onStepChange }) => {
  const windowWidth = useWindowWidth();

  return (
    <LogoContainer>
      <div className={classes.DefaultIntro}>
        <div className={classes.DefaultIntroSection}>
          <h2 className={classes.DefaultIntroHeading}>Регистрация</h2>

          <p className={classes.DefaultIntroText}>
            Поздравляем! Вы сделали первый шаг к новым карьерным возможностям!
          </p>

          <p className={classes.DefaultIntroText}>
            Теперь вам предстоит пройти небольшой тест по профориентации. Он
            поможет определить ваши сильные стороны и подобрать наиболее
            подходящие курсы для успешного перехода в новую профессию.
          </p>
        </div>

        <Button
          size={windowWidth >= 768 ? "large" : "medium"}
          color="inverse"
          isFullWidth
          onClick={() => onStepChange(0)}
        >
          Приступить к тесту
        </Button>
      </div>
    </LogoContainer>
  );
};

export default DefaultIntro;
