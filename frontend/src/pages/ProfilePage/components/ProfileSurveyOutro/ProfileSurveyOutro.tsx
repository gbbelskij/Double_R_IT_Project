import { useNavigate } from "react-router";

import { useWindowSize } from "@hooks/useWindowSize";

import Button from "@components/Button/Button";
import LogoContainer from "@components/LogoContainer/LogoContainer";
import BackgroundElements from "@components/BackgroundElements/BackgroundElements";

import classes from "./ProfileSurveyOutro.module.css";

const ProfileSurveyOutro: React.FC = () => {
  const navigate = useNavigate();
  const { isMobile } = useWindowSize();

  return (
    <LogoContainer>
      <div className={classes.ProfileSurveyOutro}>
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
          onClick={() => navigate("/profile")}
        >
          Вернуться в профиль
        </Button>

        <BackgroundElements />
      </div>
    </LogoContainer>
  );
};

export default ProfileSurveyOutro;
