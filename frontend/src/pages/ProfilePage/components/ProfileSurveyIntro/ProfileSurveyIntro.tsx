import { forwardRef } from "react";
import { useWindowSize } from "@hooks/useWindowSize";

import Button from "@components/Button/Button";
import LogoContainer from "@components/LogoContainer/LogoContainer";

import { Intro } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import classes from "./ProfileSurveyIntro.module.css";

const ProfileSurveyIntro = forwardRef(({ onStepChange, onLogoClick }, ref) => {
  const { isMobile } = useWindowSize();

  return (
    <LogoContainer onLogoClick={onLogoClick}>
      <div className={classes.ProfileSurveyIntro} ref={ref}>
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
      </div>
    </LogoContainer>
  );
}) as Intro;

export default ProfileSurveyIntro;
