import { useWindowSize } from "@hooks/useWindowSize";

import Button from "@components/Button/Button";
import LogoContainer from "@components/LogoContainer/LogoContainer";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import { IntroProps } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import classes from "./DefaultIntro.module.css";

const DefaultIntro: React.FC<IntroProps> = ({ onStepChange }) => {
  const { isMobile } = useWindowSize();

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
          size={isMobile ? "medium" : "large"}
          color="inverse"
          isFullWidth
          onClick={() => onStepChange(0)}
        >
          Приступить к тесту
        </Button>

        <BackgroundElements />
      </div>
    </LogoContainer>
  );
};

export default DefaultIntro;
