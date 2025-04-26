import { BackgroundElementsProps } from "./BackgroundElements.props";

import classes from "./BackgroundElements.module.css";

const BackgroundElements: React.FC<BackgroundElementsProps> = ({
  location = "form",
}) => {
  if (location === "form") {
    return (
      <div className={classes.BackgroundElements}>
        <svg
          width="380"
          height="419"
          viewBox="0 0 380 419"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            fillRule="evenodd"
            clipRule="evenodd"
            d="M192.399 0.217016C251.705 -2.08439 315.57 13.6626 353.005 59.7174C388.336 103.183 382.672 165.474 371.31 220.323C362.275 263.934 330.848 295.861 297.883 325.807C267.125 353.75 232.657 372.549 192.399 382.854C132.396 398.213 56.3811 443.867 14.3353 398.387C-28.0591 352.531 36.5317 282.522 42.1404 220.323C46.7655 169.033 14.7568 114.25 43.7925 71.7167C76.0121 24.5199 135.296 2.43295 192.399 0.217016Z"
            fill="var(--blue-ribbon-100)"
          />
        </svg>

        <svg
          width="287"
          height="365"
          viewBox="0 0 287 365"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            fillRule="evenodd"
            clipRule="evenodd"
            d="M158.712 0.387658C206.742 -5.00605 241.367 46.9883 267.547 87.6154C288.238 119.725 284.938 158.252 285.054 196.45C285.171 234.84 291.907 275.745 268.211 305.949C242.326 338.945 200.291 352.022 158.712 357.497C109.281 364.006 48.5379 376.94 16.0038 339.159C-16.2779 301.671 8.33558 244.52 20.03 196.45C28.3151 162.395 51.6139 138.206 71.8536 109.592C99.7757 70.1168 110.663 5.78362 158.712 0.387658Z"
            fill="var(--flush-orange-100)"
          />
        </svg>
      </div>
    );
  }
};

export default BackgroundElements;
