import { Link } from "react-router";
import { AiFillGithub } from "react-icons/ai";
import useWindowWidth from "@hooks/useWindowWidth";
import Logo from "@components/Logo/Logo";

import classes from "./Footer.module.css";

const Footer: React.FC = () => {
  const windowWidth = useWindowWidth();

  return (
    <footer className={classes.Footer}>
      <div className={classes.FooterColumn}>
        <Logo hasText />

        <p className={classes.FooterText}>
          Сделано командой{" "}
          <Link
            to="https://github.com/gbbelskij/Double_R_IT_Project"
            target="_blank"
            className={classes.FooterLinkText}
          >
            Double R
          </Link>
          <br /> В рамках проекта по дисциплине "Разработка ИТ-проектов"
        </p>
      </div>

      <Link
        to="https://github.com/gbbelskij/Double_R_IT_Project"
        target="_blank"
        className={classes.FooterColumn}
      >
        <AiFillGithub size={windowWidth >= 768 ? "49px" : "35px"} />
      </Link>

      <div className={classes.FooterColumn}>
        <p className={classes.FooterText}>
          Федеральное государственное бюджетное образовательное учреждение
          высшего образования{" "}
          <Link
            to="https://mai.ru/"
            target="_blank"
            className={classes.FooterLinkText}
          >
            «Московский авиационный институт (национальный исследовательский
            университет)»
          </Link>
        </p>

        <p className={classes.FooterText}>
          <i>г. Москва, 2025</i>
        </p>
      </div>
    </footer>
  );
};

export default Footer;
