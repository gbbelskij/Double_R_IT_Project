import { useWindowSize } from "@hooks/useWindowSize";

import Button from "@components/Button/Button";
import LogoContainer from "@components/LogoContainer/LogoContainer";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import { IntroProps } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import classes from "./ProfileSurveyIntro.module.css";

const ProfileSurveyIntro: React.FC<IntroProps> = ({ onStepChange }) => {
  const { isMobile } = useWindowSize();

  return (
    <LogoContainer>
      <div className={classes.ProfileSurveyIntro}>
        <div className={classes.ProfileSurveyIntroSection}>
          <h2 className={classes.ProfileSurveyIntroHeading}>
            Tест по профориентации
          </h2>

          <p className={classes.ProfileSurveyIntroText}>
            Вам предстоит пройти небольшой тест по профориентации. Он поможет
            определить ваши сильные стороны и подобрать наиболее подходящие
            курсы для успешного перехода в новую профессию.
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

export default ProfileSurveyIntro;
