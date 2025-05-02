import { forwardRef } from "react";

import { useWindowSize } from "@hooks/useWindowSize";

import Button from "@components/Button/Button";
import LogoContainer from "@components/LogoContainer/LogoContainer";

import { Outro } from "@components/MultiStepSurvey/MultiStepSurvey.types";

import classes from "./ProfileSurveyOutro.module.css";

const ProfileSurveyOutro = forwardRef(({ onExit }, ref) => {
  const { isMobile } = useWindowSize();

  return (
    <LogoContainer>
      <div className={classes.ProfileSurveyOutro} ref={ref}>
        <div className={classes.ProfileSurveyOutroSection}>
          <h2 className={classes.ProfileSurveyOutroHeading}>
            Спасибо за ваши ответы!{" "}
          </h2>

          <p className={classes.ProfileSurveyOutroText}>
            Мы обработаем ваши новые ответы и предложим вам лучшие курсы!
          </p>
        </div>

        <Button
          size={isMobile ? "medium" : "large"}
          color="inverse"
          isFullWidth
          onClick={() => onExit()}
        >
          Вернуться в профиль
        </Button>
      </div>
    </LogoContainer>
  );
}) as Outro;

export default ProfileSurveyOutro;
